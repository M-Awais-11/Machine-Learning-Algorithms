# Machine Learning Algorithms

A collection of machine learning projects, organized by learning type. Each project has its own folder with a notebook, a README (problem, dataset, steps, results) and plots.

## Repository Structure

```
machine-learning-algorithms/
├── supervised_learning/        # regression, classification, ensemble methods (e.g. Random Forest)
├── unsupervised_learning/      # clustering, dimensionality reduction
├── deep_learning/              # neural networks, CNNs, transfer learning
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Projects

| # | Algorithm | Project | Type | Folder | Status |
|---|-----------|---------|------|--------|--------|
| 1 | Linear Regression | Salary Prediction | Supervised (Regression) | [supervised_learning/linear_regression_salary_prediction](supervised_learning/linear_regression_salary_prediction) | Done |
| 2 | Logistic Regression | Spam Classification | Supervised (Classification) | [supervised_learning/logistic_regression_spam_classification](supervised_learning/logistic_regression_spam_classification) | Done |
| 3 | K-Means | Mall Customer Segmentation | Unsupervised (Clustering) | [unsupervised_learning/kmeans_mall_customer_segmentation](unsupervised_learning/kmeans_mall_customer_segmentation) | Done |
| 4 | MobileNetV2 (CNN) | Flower Classification | Deep Learning (Transfer Learning) | [deep_learning/flower_classification_mobilenetv2](deep_learning/flower_classification_mobilenetv2) | Done |

More algorithms will be added as new projects are completed.

## Getting Started

```bash
git clone https://github.com/<your-username>/machine-learning-algorithms.git
cd machine-learning-algorithms

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

## Project Folder Template

Every project follows the same layout:

```
project_name/
├── README.md        # problem, dataset, approach, results
├── notebook.ipynb   # code with markdown explanation
├── figures/         # saved plots
└── data/            # small datasets only (link large ones in the README)
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
