from .run import main

if len(sys.argv) == 1:
    sys.argv = sys.argv+["-h"]
main()