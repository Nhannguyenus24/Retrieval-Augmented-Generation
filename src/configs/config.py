CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "pdf_chunk"
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

MIN_TOKENS = 300
MAX_TOKENS = 500
USE_TIKTOKEN = False  # set True if tiktoken is installed for accurate token counting
TAB_WIDTH = 4         # 1 tab = 4 spaces when calculating indent
INDENT_STEP = 4       # indent difference threshold (>=) to consider as new "block"

tokenc = None