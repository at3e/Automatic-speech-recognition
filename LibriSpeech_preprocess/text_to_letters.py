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
        "--trans-file",
        required=True,
        help="Path to the transcription file"
    )
    parser.add_argument(
        "--nosymb-file",
        default=None,
        help="File with ignore word list (donot tokenize), one word per line"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get words to be ignored for tokenization
    ignore_words = None
    if args.nosymb_file:
        ignore_words = []
        with open(args.nosymb_file, "r") as nosymb:
            for line in nosymb:
                ignore_words.append(line.strip())

    # Get transcriptions
    transcriptions = {}
    set_ignore_words = set(ignore_words) if ignore_words else set()
    with open(args.trans_file, encoding="utf-8") as trans:
        for line in trans:
            key, wrd_toks, ltr_toks = tokenize_line(
                line,
                set_ignore_words
            )
            transcriptions[key] = [wrd_toks, ltr_toks]

    with open(args.tsv, "r") as tsv, open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w", encoding="utf-8"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w", encoding="utf-8"
    ) as wrd_out:
        root = next(tsv).strip()
        print('root', root)
        for line in tsv:
            line = line.strip()
            dir = os.path.dirname(line)
            # Get the key / utt-id
            key = ".".join(os.path.basename(line).split(".")[:-1])

            assert key in transcriptions

            wrds, ltrs = transcriptions[key]
            print(wrds, file=wrd_out)
            print(ltrs, file=ltr_out)


if __name__ == "__main__":
    main()

