# Effort vs. Size with Contingency Analysis

## Overview

This Python script visualizes the relationship between project size (measured in function points) and effort (measured in days) using historical data. It provides insights into best-case, worst-case, and likely effort scenarios for given function points. Additionally, it calculates contingency values and total project budgets using exponential decay models, enabling project managers and developers to make informed decisions about resource allocation and risk management.

The script generates two plots:
1. **Historical Size vs. Effort**: A scatter plot with a linear trendline showing historical data and calculated effort scenarios.
2. **Total Project Budget**: A plot showing the relationship between effort, contingency, and total budget.

## Features

- Visualizes historical size vs. effort data.
- Calculates best-case, worst-case, and likely effort scenarios.
- Computes adjusted effort scenarios based on configurable offsets.
- Models contingency using exponential decay.
- Provides interactive crosshair annotations and click-based scenario analysis.

## Requirements

- Python 3.6+
- Required Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`

## Setting Up the Environment

### 1. Clone the Repository

Clone the repository containing this script to your local machine:

```bash
git clone <repository-url>
cd <repository-folder>