import glob
import os

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{schler2006effects,
    title={Effects of age and gender on blogging.},
    author={Schler, Jonathan and Koppel, Moshe and Argamon, Shlomo and Pennebaker, James W},
    booktitle={AAAI spring symposium: Computational approaches to analyzing weblogs},
    volume={6},
    pages={199--205},
    year={2006}
}
"""

_DESCRIPTION = """\
The Blog Authorship Corpus consists of the collected posts of 19,320 bloggers gathered from blogger.com in August 2004. The corpus incorporates a total of 681,288 posts and over 140 million words - or approximately 35 posts and 7250 words per person.
Each blog is presented as a separate file, the name of which indicates a blogger id# and the blogger’s self-provided gender, age, industry and astrological sign. (All are labeled for gender and age but for many, industry and/or sign is marked as unknown.)
All bloggers included in the corpus fall into one of three age groups:
- 8240 "10s" blogs (ages 13-17),
- 8086 "20s" blogs (ages 23-27),
- 2994 "30s" blogs (ages 33-47).
For each age group there are an equal number of male and female bloggers.
Each blog in the corpus includes at least 200 occurrences of common English words. All formatting has been stripped with two exceptions. Individual posts within a single blogger are separated by the date of the following post and links within a post are denoted by the label urllink.
The corpus may be freely used for non-commercial research purposes.
"""
_URL = "https://lingcog.blogspot.com/p/datasets.html"
_DATA_URL = "data/blogs.zip"


class BlogAuthorshipCorpus(datasets.GeneratorBasedBuilder):
    """TODO(BlogAuthorship): Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="blog_authorship_corpus",
            version=VERSION,
            description="word level dataset. No processing is needed other than replacing newlines with <eos> tokens.",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "date": datasets.Value("string"),
                    "gender": datasets.Value("string"),
                    "age": datasets.Value("int32"),
                    "horoscope": datasets.Value("string"),
                    "job": datasets.Value("string"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data = dl_manager.download_and_extract(_DATA_URL)
        data_dir = os.path.join(data, "blogs")
        files = sorted(glob.glob(os.path.join(data_dir, "*.xml")))
        train_files = []
        validation_files = []

        for i, file_path in enumerate(files):
            # 95% / 5% (train / val) split
            if i % 20 == 0:
                validation_files.append(file_path)
            else:
                train_files.append(file_path)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": train_files, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"files": validation_files, "split": "validation"},
            ),
        ]

    def _generate_examples(self, files, split):
        def parse_date(line):
            # parse line to date
            return line.strip().split("<date>")[-1].split("</date>")[0]

        key = 0
        for file_path in files:
            file_name = os.path.basename(file_path)
            print("generating examples from = %s /n", file_path)
            file_id, gender, age, job, horoscope = tuple(file_name.split(".")[:-1])
            # TODO: yield also file_id?

            # Note: import xml.etree.ElementTree as etree does not work. File cannot be parsed
            # use open instead
            with open(file_path, encoding="latin_1") as f:
                date = ""
                for line in f:
                    line = line.strip()
                    if "<date>" in line:
                        date = parse_date(line)
                    elif line != "" and not line.startswith("<"):
                        if date == "":
                            logger.warning(f"Date missing for {line} in {file_name}")
                        assert date is not None, f"Date is missing before {line}"
                        yield key, {
                            "text": line,
                            "date": date,
                            "gender": gender,
                            "age": int(age),
                            "job": job,
                            "horoscope": horoscope,
                        }
                        key += 1
                    else:
                        continue

from collections import defaultdict

# # Instantiate the BlogAuthorshipCorpus class
# blog_corpus = BlogAuthorshipCorpus()

# # Download and prepare the dataset
# blog_corpus.download_and_prepare()

# Access the dataset splits
# train_dataset = blog_corpus.as_dataset(split="train")
validation_dataset = builder.as_dataset(split="validation")

# Initialize dictionaries to count samples for each class
gender_counts = defaultdict(int)
age_counts = defaultdict(int)
job_counts = defaultdict(int)
horoscope_counts = defaultdict(int)

# Iterate through the training dataset
for example in ds:
    # Access the values of each class in the example
    gender = example["gender"]
    age = example["age"]
    job = example["job"]
    horoscope = example["horoscope"]
    # Increment the counts for each class
    gender_counts[gender] += 1
    age_counts[age] += 1
    job_counts[job] += 1
    horoscope_counts[horoscope] += 1

# Print the counts for each class
print("Training Set:")
print("Gender counts:", dict(gender_counts))
print("Age counts:", dict(age_counts))
print("Job counts:", dict(job_counts))
print("Horoscope counts:", dict(horoscope_counts))

# Initialize dictionaries to count samples for each class
gender_counts = defaultdict(int)
age_counts = defaultdict(int)
job_counts = defaultdict(int)
horoscope_counts = defaultdict(int)

# Iterate through the validation dataset
for example in validation_dataset:
    # Access the values of each class in the example
    gender = example["gender"]
    age = example["age"]
    job = example["job"]
    horoscope = example["horoscope"]
    # Increment the counts for each class
    gender_counts[gender] += 1
    age_counts[age] += 1
    job_counts[job] += 1
    horoscope_counts[horoscope] += 1

# Print the counts for each class
print("Validation Set:")
print("Gender counts:", dict(gender_counts))
print("Age counts:", dict(age_counts))
print("Job counts:", dict(job_counts))
print("Horoscope counts:", dict(horoscope_counts))