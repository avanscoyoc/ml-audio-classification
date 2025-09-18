# Claude Instructions: ML Audio Classification Experiment Application

## Project Goal
Build a Kubernetes-deployable application that runs multi-factor ML experiments on audio classification models. All models have pre-trained embeddings and will be retrained in the pipeline using transfer learning for different training data sizes and multiple species, ultimately generating ROC-AUC comparison plots for each species across different models and training data sizes.

## Core Requirements

### Data Pipeline
- **Input**: Pull audio data from GCS bucket `dse-staff/soundhub/data/audio/{species_name}/data` (pos/neg subfolders)
- **Special case**: Perch model uses `dse-staff/soundhub/data/audio/{species_name}/data_5s` 
- **Output**: Write results to GCS bucket `dse-staff/soundhub/results/`
- **Species**: coyote, bullfrog, human_vocal (extensible to more)
- **Balanced sampling**: Use equal numbers of pos/neg files for training each classifier
- **Training size limits**: Maximum training size per species is constrained by `min(pos_samples, neg_samples)` to ensure balanced datasets

### ML Experiment Design
- **Models**: 5 models (birdnet, perch, vgg, mobilenet, resnet)
- **Training sizes**: 0-300 samples (configurable intervals, auto-adjusted per species based on available balanced data)
- **Cross-validation**: K-fold with set seeds for reproducibility
- **Metrics**: ROC-AUC values with confidence intervals
- **Output visualization**: Per-species charts (x=sample size, y=ROC-AUC, lines=models, error bars=CIs)

### Technical Requirements
- **Platform**: Kubernetes-ready (Docker containerized)
- **Cloud**: Google Cloud Platform integration
- **Code quality**: Implement linting (black, flake8, mypy), testing (pytest), proper logging
- **Configuration**: Environment-based config (12-factor app principles)
- **Monitoring**: Health checks, progress tracking, error handling with retries

## Architecture Suggestions
1. **Modular design**: Separate data loading, model training, evaluation, and visualization
2. **Async processing**: Use asyncio for concurrent model training
3. **Resource management**: Memory-efficient data loading, model cleanup
4. **State management**: Checkpoint intermediate results, resume capability
5. **Scalability**: Horizontal scaling via Kubernetes jobs

Feel free to redesign to achieve the most efficient and maintainable solution.

## Implementation Guidelines
- Use modern Python practices (3.10+, type hints, dataclasses/pydantic)
- Create it in a containerized docker container
- Build from scratch with clean architecture
- Create complete project structure with proper packaging
- Include setup.py/pyproject.toml, requirements files
- Implement comprehensive error handling and logging

## Deliverables
1. Complete application source code with proper structure
2. Dockerfile and Kubernetes manifests
3. Configuration files and environment setup
4. Documentation (README, API docs)
5. Tests and CI/CD pipeline suggestions

Build a production-ready solution following cloud-native best practices.