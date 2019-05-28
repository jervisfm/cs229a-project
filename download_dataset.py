
"""Simple script to download Eminst dataset. """

import tensorflow_datasets as tfds

def main():
    emnist_builder = tfds.builder("emnist")
    emnist_builder.download_and_prepare()
    emnist_info = emnist_builder.info
    print (emnist_info)


if __name__ == '__main__':
    main()