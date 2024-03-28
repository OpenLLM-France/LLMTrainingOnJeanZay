from megatron.data import indexed_dataset
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("/gpfswork/rech/knb/uyr14tk/home/lucietokenizer")

path = "/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--025_text_document"
dataset = indexed_dataset.MMapIndexedDataset(path)
for i, data in enumerate(dataset):
    if i > 10: break
    text = tokenizer.decode(data)
    text = text.replace("\n", "\\n")
    if len(text) > 110:
        print(text[:50] + f" ... {len(text)-100} ... " + text[-50:])
    else:
        print(text)

