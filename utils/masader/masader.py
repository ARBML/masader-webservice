# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Arabic Poetry Metric dataset."""


import os
import datasets
import pandas as pd

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

    def _split_generators(self, dl_manager):
        sheet_id = "1YO-Vl4DO-lnp8sQpFlcX1cDtzxFoVkCmU1PVw_ZHJDg"
        sheet_name = "filtered_clean"
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"url":url }
            ),
        ]

    def _generate_examples(self, url):
        """Generate examples."""
        # For labeled examples, extract the label from the path.
        
                
        df = pd.read_csv(url, usecols=range(35))
        df.columns.values[0] = "No."
        df.columns.values[1] = "Name"
        subsets = {}
        entry_list = []
        i = 0
        idx = 0
         
        while i < len(df.values):
            
            if i < len(df.values) - 1:
              next_entry = df.values[i+1]
            else:
              next_entry = [] 

            curr_entry = df.values[i]
            
            i+= 1
            if str(curr_entry[0]) != "nan":
                entry_list = curr_entry
                subsets = []
                
            if len(next_entry) > 0 and str(next_entry[0]) == "nan":
                subsets.append({'Name': next_entry[2], 'Dialect':next_entry[8], 'Volume':next_entry[13], 'Unit':next_entry[14]})
                continue                
            idx += 1
            masader_entry = {col:entry_list[j+1] for j,col in enumerate(df.columns[1:]) if j != 1}
            masader_entry['Year'] = int(entry_list[6])
            
            masader_entry['Subsets'] = subsets
            yield idx, masader_entry