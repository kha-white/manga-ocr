import fire

from manga_ocr.run import run


def main():
    fire.Fire(run(read_from='cli', write_to='cli'))


if __name__ == '__main__':
    main()
