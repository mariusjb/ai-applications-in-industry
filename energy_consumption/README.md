# Energy Consumption Forecasting

Public repository: [https://github.com/mariusjb/ai-applications-in-industry](https://github.com/mariusjb/ai-applications-in-industry)

In this experiment, we explore short- and medium-term energy consumption forecasting using a range of modeling approaches. We begin with a strong baseline — XGBoost, a widely adopted gradient boosting method — and challenge it with more advanced deep learning architectures, including Long Short-Term Memory, Temporal Convolutional Network, and Transformer, to evaluate their ability to capture temporal dynamics and improve predictive accuracy.

## Why Forecast Energy Consumption?
Forecasting energy consumption is essential for balancing supply and demand in power systems. Accurate predictions help ensure:
- Efficient energy production: Power plants can optimize generation schedules and reduce operating costs. Applies to energy producers and utilities.
- Smarter grid management: Grid operators can better stabilize the network, prevent overloads, and maintain frequency balance. Applies to TSOs (Transmission System Operators) and DSOs (Distribution System Operators).
- Better market participation: Energy providers and traders can place more accurate bids and reduce risk in electricity markets. Applies to energy providers, aggregators, and traders.
- Integration of renewables: Forecasting demand helps balance the variability of wind and solar, ensuring stable integration into the grid. Applies to grid operators and renewable asset managers.
- Reduced costs & emissions: Better planning minimizes the need for expensive, carbon-intensive reserve generation. Applies to all stakeholders — especially producers, grid operators, and regulators.

## Germany
Having worked for one of Germany’s largest energy producers, including its subsidiary system operators, I’ve experienced the importance of accurate demand forecasting. As Germany pursues ambitious decarbonization goals, we must not only address the rising demand driven by industrial growth and the electrification of mobility, but also ensure efficient integration of renewable energies.

In the context of volatile energy prices following the Russian invasion of Ukraine, energy independence through renewables has become more urgent than ever. However, despite record renewable capacity, significant amounts of wind and solar energy were curtailed last year due to grid congestion. [Redispatch 2.0](/energy_consumption/REDISPATCH.md), also known as Netzengpassmanagement (grid congestion management), regulates electricity feed-in and, in critical situations, can require power plants to ramp down their output (Fahrplananpassung) to ensure net stability. However, ramping down the output of power plants under Redispatch 2.0 significantly increases the cost per kilowatt-hour — often nearly doubling it. This is because, in addition to the prevailing market price for electricity, plant operators receive compensation for the curtailed generation, leading to higher overall system costs. In recent years, this has contributed to a doubling of grid fees, as the compensation costs are passed on to consumers. In turn, this leads to higher electricity prices and undermines public acceptance of renewable energy, despite its environmental benefits. Accurate forecasts that enable optimized grid operations are therefore a crucial component — and the motivator for this experiment.

***Redispatch measures led to costs of up to €2.46 billion in 2023.***


## Experiment for Assignment 1

### Data
- Hourly energy consumption data from [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data), in /data
- utils:
    - /utils/data_utils for preprocessing, feature engineering (time, lags, rolling), and sequence creation

### Exploratory Data Analysis (Brief)
- Notebook: eda.ipynb

### Approaches & Architectures
- [XGBoost](https://pypi.org/project/xgboost/)
- PyTorch implementations LSTM_ECF, TCN_ECF, and TF_ECF in /models
- utils:
    - /utils/training_utils for training and evaluation
    - /utils/visualization_utils for forecast visualization
- Notebook: forecast.ipynb (main) and multi_step_forecast.ipynb for one-step and multi-step forecasting
