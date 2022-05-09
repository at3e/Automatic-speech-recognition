import argparse
import os
import codecs


def tokenize_line(line, set_ignore_words):
    items = line.strip().split()
    key, words = items[0], items[1:]
    words = " ".join(words).lower().split()

    char_list = [
        word + " |" if word in set_ignore_words else \
            " ".join(list(word)) + " |" for word in words 
    ]
    return key, " ".join(words), " ".join(char_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument(
        "--threshold",
        required=True,
        type=int,
        help="Threshold to filter with"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    with open(args.tsv, "r") as tsv, open(
        os.path.join(args.output_dir, args.output_name + ".filtered.tsv"), "w", encoding="utf-8" 
    ) as fout:
        root = next(tsv).strip()
        print('root', root)
        print(root, file=fout)
        nfiltered = 0
        for line in tsv:
            line = line.strip()
            lsplit = line.split()
            ntokens = lsplit[-1]
            wav_path = " ".join(lsplit[:-2])
            ntokens = int(ntokens)
            if ntokens < args.threshold:
                print("{} {}".format(wav_path, ntokens), file=fout)
            else:
                nfiltered += 1

    print("Filtered out {} sentences".format(nfiltered))


if __name__ == "__main__":
    main()
