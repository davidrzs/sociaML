# NOTICE: UNDER ACTIVE DEVELOPMENT

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
![GitHub License](https://img.shields.io/github/license/davidrzs/sociaML)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/davidrzs/sociaML/ci.yml)
![PyPI - Version](https://img.shields.io/pypi/v/sociaML)



# sociaML - the Swiss Army knife for audiovisual and textual video feature extraction.

With sociaML you can extract features relevant downstream research (eg social sciences) with little knowledge machine learning or even Python.

## Attention: Currently we are Linux only.

This stems from the fact that we depend on Whisper which depends on Triton which is Linux only for the time being. 


## Getting Started

This section is under development.


## Explanation of Concepts

Assuming we have a video actors playing Hamlet by Shakespeare as seen in the image below. SociaML knowns three levels of feature collection:

1. global features
2. participant features
3. contribution features

![illustration of concepts](https://raw.githubusercontent.com/davidrzs/sociaML/main/docs/images/feature_matrix.png?token=GHSAT0AAAAAACLXMZ3H7UXHAADMIKXAKHZWZNDHXRA)

TODO
![pipeline](https://raw.githubusercontent.com/davidrzs/sociaML/main/docs/images/pipeline.png?token=GHSAT0AAAAAACLXMZ3GTH4TYFX3ETB3LZWQZNDHXRA)

TODO

## Collaborating and Getting Involved 

If you have feature requests or want to co-develop this package please do not hesitate to reach out! 


## Collaborators

Main Developers

- David Zollikofer zdavid@ethz.ch
- Loic Cabon lcabon@ethz.ch

Technical guidance by 

- [Elliott Ash](https://elliottash.com/)
- [Aniket Kesari](https://www.aniketkesari.com/)


## License

Code is licensed under the permissive MIT license. Certain modules we depend on have different licenses though!
