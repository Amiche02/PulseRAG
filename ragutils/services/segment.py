import re

class CustomSegment:
    """
    Handles Hybrid segmentation using a combination of paragraph-based, token-based, and sentence-based logic.
    """
    def __init__(self, min_tokens=50, max_tokens=200):
        """
        Initializes the CustomSegment with minimum and maximum token constraints.

        Args:
            min_tokens (int): Minimum number of tokens required in a chunk.
            max_tokens (int): Maximum number of tokens allowed in a chunk.
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def hybrid_segmentation(self, text: str) -> list:
        """
        Performs hybrid segmentation on the provided text.

        The method normalizes line breaks, splits the text into paragraphs, and further segments paragraphs
        based on token counts and sentence boundaries.

        Args:
            text (str): The input text to be segmented.

        Returns:
            list: A list of segmented text chunks.
        """
        # Normalize multiple line breaks to single
        text = re.sub(r'\n+', '\n', text)
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            tokens = self._tokenize(paragraph)

            if self.min_tokens <= len(tokens) <= self.max_tokens:
                chunks.append(paragraph.strip())
            elif len(tokens) > self.max_tokens:
                sentences = self._split_into_sentences(paragraph)
                temp_chunk = ""
                for sentence in sentences:
                    temp_tokens = self._tokenize(temp_chunk + " " + sentence)
                    if len(temp_tokens) <= self.max_tokens:
                        temp_chunk += " " + sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
            elif len(tokens) < self.min_tokens:
                current_chunk += " " + paragraph
                tokens_in_chunk = self._tokenize(current_chunk)
                if len(tokens_in_chunk) >= self.min_tokens:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_into_paragraphs(self, text: str) -> list:
        """
        Splits text into individual paragraphs.

        Args:
            text (str): The input text.

        Returns:
            list: A list of paragraphs.
        """
        paragraphs = re.split(r'\n{2,}', text)
        return [para.strip() for para in paragraphs if para.strip()]

    def _split_into_sentences(self, text: str) -> list:
        """
        Splits text into sentences using regex patterns.

        Args:
            text (str): The input text.

        Returns:
            list: A list of sentences.
        """
        return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

    def _tokenize(self, text: str) -> list:
        """
        Tokenizes the text into words using regex.

        Args:
            text (str): The input text.

        Returns:
            list: A list of tokens.
        """
        return re.findall(r'\b\w+\b', text)
