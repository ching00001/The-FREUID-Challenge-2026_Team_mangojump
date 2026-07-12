"""Official Docker entrypoint for the frozen FREUID inference system."""

from src.predict_docker import main


if __name__ == "__main__":
    main()
