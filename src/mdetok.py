from megatron.data import indexed_dataset
import transformers

# cf. https://huggingface.co/OpenLLM-France/Lucie-tokenizer-v2.4-space_prefix_all/tree/main

tokenizer = transformers.AutoTokenizer.from_pretrained("/gpfswork/rech/knb/uyr14tk/home/lucietokenizer")

# path = "/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--025_text_document"

fichs = ("/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--000_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--001_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--002_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--003_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--004_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--005_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--006_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--007_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--008_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--009_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--010_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--011_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--012_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--013_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--014_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--015_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--016_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--017_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--018_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--019_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--020_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--021_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--022_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--023_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--024_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--025_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--026_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--027_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--028_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--029_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--030_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--031_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--032_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--033_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--034_text_document","/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--035_text_document")
for path in fichs:
    dataset = indexed_dataset.MMapIndexedDataset(path)
    for i, data in enumerate(dataset):
        if True:
            s = ' '.join([str(x) for x in data])
            print("Y "+s)
        else:
            text = tokenizer.decode(data)
            text = text.replace("\n", "\\n")
            print(text)

