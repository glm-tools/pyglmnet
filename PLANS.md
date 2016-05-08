# Future directions

This is a collection of ideas for future directions of this project, mainly intended to keep track of suggestions.
The ideas may ideally be executed through funding channels such as [GSOC](https://developers.google.com/open-source/gsoc/)

### Enhancements and Differentiators
- Provide alternate penalty functions e.g. Tikhonov regularizers for the L2 part of the loss function: very useful for time series models
- Implement group lasso/ group elastic net: very useful to regularize in groups
- Implement SGD variants by potentially integrating with `sklearn`: useful for big data
- Make a contribution to Spark's MLLib: useful for streaming data

### Use cases
- Interface with companion packages such as proposed `Spykes` and `NeuroSTRF` to demonstrate use cases for neural data
- Interface with big functional neural data as Jeremy Freeman's mouse or zerbrafish datasets.
