freq = {}
with open('${runconfig}/finetune_data/train.ltr', 'w') as outfile:
  with open('data/lists/pmk.gsw.train.lst') as infile:
    for line in infile:
      sentence = line.strip().split()
      for word in sentence[3:]:
        outfile.write(" ".join(list(word)) + " | ")
        for w in word + "|":
          freq[w] = freq[w] + 1 if w in freq else 1
      outfile.write('\n')

for dataset in ['val', 'test']:
  with open('${runconfig}/finetune_data/' + dataset + '.ltr', 'w') as outfile:
    with open('data/lists/pmk.gsw.' + dataset + '.lst') as infile:
      for line in infile:
        sentence = line.strip().split()
        for word in sentence[3:]:
          outfile.write(" ".join(list(word)) + " | ")
        outfile.write('\n')

with open('${runconfig}/finetune_data/dict.ltr.txt', 'w') as outfile:
    for token, f in sorted(freq.items(), key=lambda item: item[1], reverse=True):
      outfile.write(token + " " + str(f) + "\n")
