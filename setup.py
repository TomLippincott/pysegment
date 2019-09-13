from distutils.core import setup

setup(
    name="pysegment",
    version="0.4",
    packages=["pysegment"],
    license="Creative Commons Attribution-Noncommercial-Share Alike license",
    long_description="Tools for unsupervised segmentation and morphology",
    package_dir={"pysegment" : "src"},
    install_requires=["nltk"],
    package_data={"" : ["grammar_templates/unigram.txt", 
                        "grammar_templates/simple_prefix_suffix.txt", 
                        "grammar_templates/simple_agglutinative.txt"]},
    scripts=["scripts/pysegment"],
)
