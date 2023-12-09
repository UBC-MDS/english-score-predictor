## English Language Learning Ability Prediction Changelog
### Release: [3.0.0](https://github.com/UBC-MDS/522-workflows-group-18/tree/3.0.0)

**Fixed bugs:**

- [a91eb1e](https://github.com/UBC-MDS/522-workflows-group-18/commit/a91eb1ecea2219b42b30e21bcca8a1023e690cf6): Implemented Github Actions Workflow using DockerHub Image 
- [402094b](https://github.com/UBC-MDS/522-workflows-group-18/commit/402094b8f906e4f92f0cd9ec749b27f19481eafd): Addressed test case failures in helper function: `tests/test-plt_regr_pred.py`, Fixed return to be Axes object and added unittest rather than pytest
- [b9a06f5](https://github.com/UBC-MDS/522-workflows-group-18/commit/b9a06f5875ca940933eeb148d9a5aa6dd6568d55): Addressed test case failures in helper function: `tests/show_feat_coeff.py`, Fixed return to get feature names out from X_train rather than X_enc and added unittest rather than pytest and adjusted parameters for FunctionTransformer (feature_names_out='one-to-one") so that we could get all feature names 
- [\#46](https://github.com/UBC-MDS/522-workflows-group-18/pull/47/commits): fixed dropping of some additional question level columns that wouldn't be used in the analysis but were being dropped outside the transformer.


**Implemented enhancements:**

- [a290eb5](https://github.com/UBC-MDS/522-workflows-group-18/commit/a290eb55db4eba745c33db6f36c8456a35647503): Changed from specifying packages in docker file to using `environment.yml` file 
- [908b58a](https://github.com/UBC-MDS/522-workflows-group-18/pull/101/commits/908b58a4099f6f096f24eae080992fc74165ae95) and [c476b28](https://github.com/UBC-MDS/522-workflows-group-18/pull/101/commits/c476b28c8e5e79e7187c707f4e85415da07485ca): Added a reference to final report (Background part)
- [18b36c4](https://github.com/UBC-MDS/522-workflows-group-18/pull/99/commits/18b36c4ffed23ca2d4e8848275de6c9f1dd5e417): Updated license file with creative license and database
- [\#92](https://github.com/UBC-MDS/522-workflows-group-18/pull/92): Formatted the ReadME to be more readable and added contact channel
- [\#103](https://github.com/UBC-MDS/522-workflows-group-18/pull/103/files): Testing automation, and Added README info to test the helper function tests easily 
- [\#105](https://github.com/UBC-MDS/522-workflows-group-18/pull/105/files): Added more EDA plots and made the previous ones more clear for addition in report
- [\#107](https://github.com/UBC-MDS/522-workflows-group-18/pull/107/files): Added other scoring metrics on test results 
- [](): Fixed the Actual vs Predicted plots overplotting issue 
- [](): Fixed wordings in report to make it more clear 
- 

**Closed issues during Week 4:**

- [\#58](https://github.com/UBC-MDS/522-workflows-group-18/issues/58): Previous Improvements that we discussed could be implemented and fixed  
- [\#96](https://github.com/UBC-MDS/522-workflows-group-18/issues/96): Peer Review Comments that were addressed 
- [\#108](https://github.com/UBC-MDS/522-workflows-group-18/issues/108): Addressed all concerns regarding final report from peer review and group feedback
- [\#98](https://github.com/UBC-MDS/522-workflows-group-18/issues/98): Fixed concerns regarding license