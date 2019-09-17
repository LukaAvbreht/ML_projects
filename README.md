# ML_projects

## Questions 1

1. What is the problem of interest? (describe what your data is about in general terms, i.e. “to someone who knows nothing of machine learning”)
    * The data is about airbnb properties in NYC.
    * Each observation contains useful information like:
        * Price
        * Coordinates
        * Neighbourhood name
        * Availability
        * Name of property
        * ...

2. Who made the data and why? Did they, or somebody else, work with the data and report results? If so, what were their results?
    * Published by Airbnb for public use
    * No one did anything with the data so far (as we are aware of)

3. What is the primary machine learning modelling aim? (Is it primarily a classification, a regression, a clustering, an association mining, or an anomaly detection problem?)
    * clustering and some regression

4. Which attributes are relevant when carrying out a classification, a regression, a clustering, an association mining, and an anomaly detection? Specifically: Which attribute do you wish to explain in the regression based on which other attributes? Which class label will you predict based on which other attributes in the classification task?
    * Price
    * Neighbourhood name
    * Name of property (words used)
    * Frequency/last/number of reviews
    * Listings per host
    * Availability per 365

5. Are there any data issues? Either directly reported in the accompanying dataset description or apparent by inspection of the data? (such as missing values or incorrect/corrupted values)
    * We don't know the date on which the data was picked (prices are the main concern)
    * Some of the properties are outdated, they are not on airbnb anymore