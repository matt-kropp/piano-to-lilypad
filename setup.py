from setuptools import setup, find_packages

setup(
    name="piano_to_lilypond",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "librosa>=0.9.2",
        "soundfile>=0.12.1",
        "pretty_midi>=0.2.10",
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "psutil>=5.8.0"
    ],
    entry_points={
        'console_scripts': [
            'p2l_train=piano_to_lilypond.train:main',
            'p2l_infer=piano_to_lilypond.infer:main'
        ]
    }
)