from distutils.core import setup

setup(
    name='pyAudioAnalysis3',
    description='Python 3 port of pyAudioAnalysis by tyiannak',
    long_description=open('README.md').read(),
    version='0.1dev',
    packages=['pyaudioanalysis'],
    package_data={'pyaudioanalysis': ['resources/*']},
    url='https://github.com/gabrielecalvo/pyAudioAnalysis',
    license='LICENSE.md',
    install_requires=["numpy", "matplotlib", "scipy", "simplejson", "scikit-learn", "eyed3", "pydub", "hmmlearn"],
    include_package_data=True,
)