# Documentation

This directory contains documentation for the ML-Enhanced Pairs Trading project.

## Contents

- **methodology.md**: Detailed explanation of the pairs trading methodology and ML approaches
- **data_sources.md**: Information about data sources and how to acquire data
- **api_reference.md**: API reference for the project modules
- **tutorials/**: Step-by-step tutorials for using the system

## Getting Started

For a quick start guide, see the main [README.md](../README.md) in the project root.

## Methodology Overview

Pairs trading is a market-neutral strategy that involves identifying two historically correlated assets and taking opposite positions when their prices diverge, expecting them to converge back to their historical relationship.

This project enhances traditional pairs trading with machine learning techniques to:
1. Identify cointegrated pairs more accurately
2. Predict optimal entry and exit points
3. Adapt to changing market conditions

## Key Concepts

- **Cointegration**: Statistical property indicating a long-term relationship between two time series
- **Spread**: The difference between the prices of two stocks in a pair
- **Z-score**: Normalized measure of how far the current spread is from its mean
- **Mean Reversion**: The tendency of the spread to return to its historical mean
