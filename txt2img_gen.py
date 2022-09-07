import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pygame
from PIL import Image
from pygame import freetype

from utils import overlapping_n_grams

DEFAULT_FONT_SIZE = 10
DEFAULT_PAD_SIZE = 3
DEFAULT_PPB = 20
MAX_SEQ_LENGTH = 7
MAX_PIXELS_LEN = MAX_SEQ_LENGTH * DEFAULT_PPB


logger = logging.getLogger(__name__)

SUPPORTED_INPUT_TYPES = [str, Tuple[str, str], List[str]]


@dataclass
class Encoding:
    """
    Dataclass storing renderer outputs
    Args:
        pixel_values (`numpy.ndarray`):
            A 3D numpy array containing the pixel values of a rendered image
        sep_patches (`List[int]`):
            A list containing the starting indices (patch-level) at which black separator patches were inserted in the
            image.
        num_text_patches (`int`):
            The number of patches in the image containing text (excluding the final black sep patch). This value is
            e.g. used to construct an attention mask.
        word_starts (`List[int]`, *optional*, defaults to None):
            A list containing the starting index (patch-level) of every word in the rendered sentence. This value is
            set when rendering texts word-by-word (i.e., when calling a renderer with a list of strings/words).
        offset_mapping (`List[Tuple[int, int]]`, *optional*, defaults to None):
            A list containing `(char_start, char_end)` for each image patch to map between text and rendered image.
        overflowing_patches (`List[Encoding]`, *optional*, defaults to None):
            A list of overflowing patch sequences (of type `Encoding`). Used in sliding window approaches, e.g. for
            question answering.
        sequence_ids (`[List[Optional[int]]`, *optional*, defaults to None):
            A list that can be used to distinguish between sentences in sentence pairs: 0 for sentence_a, 1 for
            sentence_b, and None for special patches.
    """

    pixel_values: Union[List[np.ndarray], np.ndarray]
    sep_patches: List[int]
    num_text_patches: int
    word_starts: Optional[List[int]] = None
    offset_mapping: Optional[List[Tuple[int, int]]] = None
    overflowing_patches: Optional[List] = None
    sequence_ids: Optional[List[Optional[int]]] = None


class TextImageGenerator:

    def __init__(
        self,
        font_file: str,
        dpi: int = 150,
        background_color: str = "white",
        font_color: str = "black",
        font_size: int = DEFAULT_FONT_SIZE,
        pad_size: int = DEFAULT_PAD_SIZE,
        pixels_per_patch: int = DEFAULT_PPB,
        max_seq_length: int = MAX_SEQ_LENGTH,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.font_file = font_file
        self.font_size = font_size
        self.font_color = font_color
        self.background_color = background_color

        self.pixels_per_patch = pixels_per_patch
        self.max_seq_length = max_seq_length
        self.pad_size = pad_size
        self.pad_left = pad_size
        self.pad_right = pad_size
        self.pad_top = pad_size
        self.pad_bottom = pad_size

        self.dpi = dpi
        freetype.init()
        freetype.set_default_resolution(dpi)

        self.font = None
        self.load_font()

    @property
    def max_pixels_len(self):
        return self.max_seq_length * self.pixels_per_patch

    def __getstate__(self):
        """
        Returns the state dict of the renderer without the loaded font to make it picklable
        Returns:
            The state dict of type `Dict[str, Any]`
        """

        return {
            "font_file": self.font_file,
            "dpi": self.dpi,
            "background_color": self.background_color,
            "font_color": self.font_color,
            "font_size": self.font_size,
            "pad_size": self.pad_size,
            "pixels_per_patch": self.pixels_per_patch,
            "max_seq_length": self.max_seq_length,
        }

    def __setstate__(self, state_dict: Dict[str, Any]) -> None:
        """
        Sets the state dict of the renderer, e.g. from a pickle
        Args:
            state_dict (`Dict[str, Any]`):
                The state dictionary of a `PyGameTextRenderer`, containing all necessary 
                and optional fields to initialize a `PyGameTextRenderer`.
        """

        self.__init__(**state_dict)

    def _get_offset_to_next_patch(self, x: int) -> int:
        """
        Get the horizontal position (offset) where the next patch begins,
        based on how many pixels a patch contains the maximum width.

        Args:
            x (`int`):
                The horizontal position from where the next patch is to be found

        Returns:
            The starting position of the next patch (offset) of type `int`
        """

        return min(
            math.ceil(x / self.pixels_per_patch) * self.pixels_per_patch,
            self.max_pixels_len - self.pixels_per_patch
        )

    def _get_empty_surface(self) -> pygame.Surface:
        """
        Create and return an empty surface that we will later render text to.

        Returns:
            The blank surface of type (`~pygame.Surface`)
        """

        frame = (self.max_pixels_len, self.pixels_per_patch)
        surface = pygame.Surface(frame)
        surface.fill(pygame.color.THECOLORS[self.background_color])
        return surface

    def _draw_black_patch(
        self,
        offset: int,
        surface: pygame.Surface
    ) -> pygame.Surface:
        """
        Draws a black separator patch on a surface a horizontal offset, i.e.
        the black patch begins <offset> pixels to the right from the 
        beginning of the surface.

        Args:
            offset (`int`):
                The horizontal starting position of the black black patch on
                the surface (in pixels)
            surface (`~pygame.Surface`):
                The surface that the black patch is drawn on

        Returns:
            A surface of type `~pygame.Surface` with the black patch drawn on it
        """

        sep_rect = pygame.Rect(
            offset,
            0,
            self.pixels_per_patch,
            self.pixels_per_patch)
        pygame.draw.rect(surface, self.font.fgcolor, sep_rect)
        return surface

    def _render_words_to_surface(self, words: List[str]) -> List[Encoding]:
        """
        Renders a list of words to a new surface, i.e. treats each word
        as a sentence, and performs _render_single_sentence for each word.

        Args:
            words (`List[str]`):
                The list of words to be rendered
        Returns:
            An encoding of type `Encoding` containing the rendered words and metadata
        """

        encodings_list = []

        for word in words:
            encoding = self._render_text_to_surface(word)
            encodings_list.append(encoding)

        return encodings_list

    def _render_single_sentence(
        self,
        sentence: str,
        offset: int,
        surface: pygame.Surface
    ) -> Tuple[pygame.Surface, int]:
        """
        Renders a single sentence to a surface with a horizontal offset, i.e. 
        the rendered sentence begins <offset> pixels to the right from the
        beginning of the surface, and centers the rendered text vertically on
        the surface.
        """
        text_surface, rect = self.font.render(sentence, self.font.fgcolor)
        rect.midleft = (offset, surface.get_height() / 2)
        surface.blit(text_surface, rect)
        return surface, rect.width

    def _render_text_to_surface(self, text: str) -> Encoding:
        """
        Renders a single piece of text, e.g. a sentence or paragraph, to a
        surface and keeps track of how many patches in the rendered surface
        contain text, i.e. are neither blank nor blank separator patches.

        Args:
            text (`str`):
                The piece of text to be rendered

        Returns:
            An encoding of type `Encoding` containing the rendered text
            and metadata
        """

        surface = self._get_empty_surface()
        sep_patches = []

        offset = 2

        # Render text
        surface, text_width = self._render_single_sentence(
            sentence=text,
            offset=offset,
            surface=surface
        )

        # Offset is left padding + rendered width of first sentence + 2 (padding)
        offset = self._get_offset_to_next_patch(2 + text_width + 2)

        # Draw black rectangle on surface as separator patch
        surface = self._draw_black_patch(offset=offset, surface=surface)
        sep_patches.append(offset // self.pixels_per_patch)

        num_text_patches = math.ceil(offset / self.pixels_per_patch)

        encoding = Encoding(
            pixel_values=self.get_image_from_surface(surface),
            num_text_patches=num_text_patches,
            sep_patches=sep_patches
        )

        return encoding

    @staticmethod
    def get_image_from_surface(surface: pygame.Surface) -> np.ndarray:
        """
        Transforms a surface containing a rendered image into a numpy array.

        Args:
            surface (`pygame.Surface`):
                The pygame surface containing the rendered text

        Returns:
            An image of type `np.ndarray` of size
            [self.pixels_per_patch, self.max_pixels_len]
        """

        image = pygame.surfarray.pixels3d(surface=surface)
        image = image.swapaxes(0, 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def __call__(
        self,
        text: Union[str, Tuple[str, str], List[str]],
    ) -> Encoding:
        """
        Render a piece of text to a surface, convert the surface into an image
        and return the image along with metadata (the number of patches containing
        text and, when rendering a list of words, the patch indices at which each
        word starts).

        Args:
            text (`str` or `Tuple[str, str]` or `List[str]`):
                The text to be rendered

        Returns:
            An encoding of type `Encoding` containing the rendered text input and metadata
        """

        if isinstance(text, str):
            rendering_fn = self._render_text_to_surface
        elif isinstance(text, list):
            rendering_fn = self._render_words_to_surface
        else:
            raise TypeError(
                f"{self.__class__.__name__} does not support inputs of type {type(text)}. "
                f"Support types are {SUPPORTED_INPUT_TYPES}"
            )

        # Render input
        encoding = rendering_fn(text)

        return encoding

    def load_font(self) -> None:
        """
        Loads the font from specified font file with specified font size and color.
        """

        logger.info(f"Loading font from {self.font_file}")

        font = freetype.Font(self.font_file, self.font_size)
        font.style = freetype.STYLE_NORMAL
        font.fgcolor = pygame.color.THECOLORS[self.font_color]

        self.font = font


def main(args):

    gen = TextImageGenerator(
        font_file=args.font_file
    )
    encoding = gen(args.text)

    if isinstance(encoding, List):
        for idx in range(len(encoding)):
            arr = encoding[idx].pixel_values
            np2pil = Image.fromarray(arr)
            np2pil.save(
                f"/home/asohn3/baraslab/hla/Data/pep2imgtxt/pil_{idx+1}.png")
    else:
        arr = encoding.pixel_values
        np2pil = Image.fromarray(arr)
        np2pil.save(
            f"/home/asohn3/baraslab/hla/Data/pep2imgtxt/pil_1.png")


if __name__ == "__main__":
    import argparse

    inp = overlapping_n_grams("AKHRGPAHDLALEPDSP", n_gram_size=9)
    # inp = "AKHRGPAHDLALEPDSP"
    # inp = pd.read_pickle(
    #     '/home/asohn3/baraslab/hla/Data/regression_subset/TCGA-2F-A9KP-01A-11D-A38G-08_final_nonsyn_p9.pkl'
    # )

    parser = argparse.ArgumentParser()
    parser.add_argument("--font-file", type=str,
                        default="/home/asohn3/baraslab/hla/Fonts/NotoMono-Regular.ttf")
    parser.add_argument("--text", type=str, default=inp)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
