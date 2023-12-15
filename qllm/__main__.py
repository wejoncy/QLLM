from .run import main
import sys

if len(sys.argv) == 1:
    sys.argv = sys.argv+["-h"]
main()