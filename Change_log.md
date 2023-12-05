## English Language Learning Ability Prediction Changelog
### Release: [3.0.0](https://github.com/UBC-MDS/522-workflows-group-18/tree/2.0.0)
***Update above later to link to the final release for Milestone 4***

**Fixed bugs:**

- [a91eb1e](https://github.com/UBC-MDS/522-workflows-group-18/commit/a91eb1ecea2219b42b30e21bcca8a1023e690cf6): Implemented Github Actions Workflow using DockerHub Image 
- [402094b](https://github.com/UBC-MDS/522-workflows-group-18/commit/402094b8f906e4f92f0cd9ec749b27f19481eafd): Addressed test case failures in helper function: `tests/test-plt_regr_pred.py`, Fixed return to be Axes object and added unittest rather than pytest
- [b9a06f5](https://github.com/UBC-MDS/522-workflows-group-18/commit/b9a06f5875ca940933eeb148d9a5aa6dd6568d55): Addressed test case failures in helper function: `tests/show_feat_coeff.py`, Fixed return to get feature names out from X_train rather than X_enc and added unittest rather than pytest and adjusted parameters for FunctionTransformer (feature_names_out='one-to-one") so that we could get all feature names 
- [\#46](https://github.com/UBC-MDS/522-workflows-group-18/pull/47): fixed dropping of some additional question level columns that wouldn't be used in the analysis but were being dropped outside the transformer.

**Implemented enhancements:**

- [a290eb5](https://github.com/UBC-MDS/522-workflows-group-18/commit/a290eb55db4eba745c33db6f36c8456a35647503): Changed from specifying packages in docker file to using `environment.yml` file 


**Merged pull requests:**
To be updated
- Fixed target for SPM - Package.swift file [\#46](https://github.com/UBC-MDS/522-workflows-group-18/pull/47) ([noorulain17](https://github.com/noorulain17))


**Closed issues:**
To be updated
- ActionSheetDatePicker still crash in Xcode12 when set the datePickerStyle  [\#498](https://github.com/skywinder/ActionSheetPicker-3.0/issues/498)