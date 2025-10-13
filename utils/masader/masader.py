import datasets
from glob import glob 
import json 
import zipfile
from random import shuffle

_DESCRIPTION = """\
Masader is the largest public catalogue for Arabic NLP datasets, which consists of more than 200 datasets annotated with 25 attributes. 
"""

_CITATION = """\
@misc{alyafeai2021masader,
      title={Masader: Metadata Sourcing for Arabic Text and Speech Data Resources}, 
      author={Zaid Alyafeai and Maraim Masoud and Mustafa Ghaleb and Maged S. Al-shaibani},
      year={2021},
      eprint={2110.06744},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
      }
"""


class MasaderConfig(datasets.BuilderConfig):
    """BuilderConfig for Masader."""

    def __init__(self, **kwargs):
        """BuilderConfig for MetRec.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MasaderConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class Masader(datasets.GeneratorBasedBuilder):
    """Masaderdataset."""

    BUILDER_CONFIGS = [
        MasaderConfig(
            name="plain_text",
            description="Plain text",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'Name': datasets.Value("string"),
                    'Subsets': [{'Name':datasets.Value("string"), 
                                 'Dialect':datasets.Value("string") , 
                                 'Volume':datasets.Value("string") , 
                                 'Unit':datasets.Value("string")}],
                    'HF Link': datasets.Value("string"),
                    'Link': datasets.Value("string"),
                    'License': datasets.Value("string"),
                    'Year': datasets.Value("int32"),
                    'Language': datasets.Value("string"),
                    'Dialect': datasets.Value("string"),
                    'Domain': datasets.Value("string"),
                    'Form': datasets.Value("string"),
                    'Collection Style': datasets.Value("string"),
                    'Description': datasets.Value("string"),
                    'Volume': datasets.Value("string"),
                    'Unit': datasets.Value("string"),
                    'Ethical Risks': datasets.Value("string"),
                    'Provider': datasets.Value("string"),
                    'Derived From': datasets.Value("string"),
                    'Paper Title': datasets.Value("string"),
                    'Paper Link': datasets.Value("string"),
                    'Script': datasets.Value("string"),
                    'Tokenized': datasets.Value("string"),
                    'Host': datasets.Value("string"),
                    'Access': datasets.Value("string"),
                    'Cost': datasets.Value("string"),
                    'Test Split': datasets.Value("string"),
                    'Tasks': datasets.Value("string"),
                    'Venue Title': datasets.Value("string"),
                    'Citations': datasets.Value("string"),
                    'Venue Type': datasets.Value("string"),
                    'Venue Name': datasets.Value("string"),
                    'Authors': datasets.Value("string"),
                    'Affiliations': datasets.Value("string"),
                    'Abstract': datasets.Value("string"),
                    'Added By': datasets.Value("string"), 
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/arbml/Masader",
            citation=_CITATION,)
	
    def extract_all(self, dir):
        zip_files = glob(dir+'/**/**.zip', recursive=True)
        for file in zip_files:
            with zipfile.ZipFile(file) as item:
                item.extractall('/'.join(file.split('/')[:-1])) 



    def _split_generators(self, dl_manager):
        branch = "main"
        url = [f'https://github.com/ARBML/masader/archive/{branch}.zip']
        downloaded_files = dl_manager.download_and_extract(url)
        self.extract_all(downloaded_files[0])
        all_files = sorted(glob(downloaded_files[0]+f'/masader-{branch}/datasets/**.json'))
        shuffle(all_files)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'filepaths':{'inputs':all_files} })]
    
    def _generate_examples(self, filepaths):        
        for idx,filepath in enumerate(filepaths['inputs']):
            with open(filepath, 'r') as f:
                data = json.load(f)
                # cast list items to string
                for key in data:
                    if isinstance(data[key], list) and key != 'Subsets':
                        data[key] = ','.join(data[key])
            yield idx, data