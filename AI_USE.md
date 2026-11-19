# AI Use Disclosure

**Project:** *Gaia DR3 Stellar Clustering: An End-to-End Machine Learning Pipeline*  

## 1. Purpose of AI assistance

AI was used as a technical support tool during the development of this project. It provided explanations, suggested implementation approaches and revised some code and configuration, helped troubleshoot errors, and assisted with some technical documentation.

The assistance was used interactively. I supplied the project context, requirements, code or command output where relevant; I then ran commands and tests in my own project environment, examined the results, and made decisions about which proposed changes to adopt.

## 2. Areas in which AI assistance was used

### Conceptual and technical explanations

The AI was used to explain and clarify certain machine-learning and software-engineering concepts relevant to the project. This includes clustering evaluation, model inference, REST APIs, Docker, Git and GitHub Releases.

### Code and configuration support

The AI provided code suggestions and helped refine or troubleshoot parts of the project's Python scripts and supporting implementation. This included assistance related to the data pipeline, model-training workflow and artifact-verification procedure.

It also provided some drafting assistance for the FastAPI application and its API requirements, as well as the Dockerfile and Docker Compose configuration.

### Testing and troubleshooting support

AI helped interpret certain error messages and test outputs. These suggestions were not treated as proof that the implementation worked. The relevant scripts, API requests, artifact checks and Docker deployment were run in the project environment, and their actual outputs were examined before results were recorded.

### Documentation support

It helped identify and incorporate the Gaia DR3 acknowledgement and bibliographic references from the official Gaia documentation.

## 3. Work performed and decisions retained by the student

I directed the project work and made the decisions about the project scope, dataset and intended workflow. I operated the development environment and executed the data-processing, training, verification, API-testing, Docker and Git commands on my computer.

I examined the outputs produced in that environment, reported errors and unexpected behaviour, chose whether to apply suggested changes, and checked the resulting behaviour. Examples of work I executed and checked include:

- Running the data pipeline and checking its processed outputs and feature consistency with the notebook.
- Running the model-training workflow and examining the resulting clustering and evaluation outputs.
- Loading the saved model artifacts and carrying out sample inference checks without refitting the models.
- Starting the FastAPI service and testing its endpoints, including a prediction request using a record from the Gaia catalogue.
- Building and starting the Docker deployment, investigating its startup logs, applying the Numba cache configuration change, and confirming that the container became healthy and that API requests returned successful responses.
- Managing the Git repository, reviewing the files to be committed, pushing project changes, and publishing the versioned release and its downloadable artifacts.

The reported dataset counts, model outputs, evaluation values, API responses and deployment results are based on the project's executed scripts, saved artifacts and observed test outputs. They were not invented by the AI.

## 4. Limits and verification of AI-generated material

AI-generated explanations, code, commands and documentation can contain errors, omissions or unsuitable assumptions. I therefore used the actual project environment and observed outputs to check proposed implementation changes. In particular, successful API and Docker behaviour was established through execution and testing, rather than inferred solely from AI-generated code.

## 5. Declaration

I acknowledge that AI made a meaningful contribution as a development assistant, moreso revising some implementation code and deployment configuration, as well as providing certain explanations and troubleshooting support. This disclosure does not claim that the work was completed without AI assistance, nor that every implementation decision or validation was performed by the AI tool.

I have reviewed and tested the project components described above in my development environment and accept responsibility for the accuracy, integrity and final content of the submitted work.
