.. DebiasED documentation master file, created by
   sphinx-quickstart on Sun Jul 20 10:20:32 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DebiasED documentation
======================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


Welcome to your_project's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Crossvalidation modules

   debiased.crossvalidation.scorers.binary_scorer
   debiased.crossvalidation.scorers.fairness_binary_scorer

.. toctree::
   :maxdepth: 3
   :caption: Mitigation modules - Inprocessing

   debiased.mitigation.inprocessing.chakraborty_in
   debiased.mitigation.inprocessing.chen
   debiased.mitigation.inprocessing.gao
   debiased.mitigation.inprocessing.grari2
   debiased.mitigation.inprocessing.islam
   debiased.mitigation.inprocessing.kilbertus
   debiased.mitigation.inprocessing.liu
   debiased.mitigation.inprocessing.zafar

.. toctree::
   :maxdepth: 3
   :caption: Mitigation modules - Preprocessing

   debiased.mitigation.preprocessing.alabdulmohsin
   debiased.mitigation.preprocessing.calders
   debiased.mitigation.preprocessing.chakraborty
   debiased.mitigation.preprocessing.cock
   debiased.mitigation.preprocessing.dablain
   debiased.mitigation.preprocessing.iosifidis_resampledattribute
   debiased.mitigation.preprocessing.iosifidis_resampletarget
   debiased.mitigation.preprocessing.iosifidis_smoteattribute import
   debiased.mitigation.preprocessing.iosifidis_smotetarget
   debiased.mitigation.preprocessing.lahoti
   debiased.mitigation.preprocessing.li
   debiased.mitigation.preprocessing.zelaya_over
   debiased.mitigation.preprocessing.zelaya_smote
   debiased.mitigation.preprocessing.zelaya_over
   debiased.mitigation.preprocessing.zemel

.. toctree::
   :maxdepth: 3
   :caption: Predictors

   debiased.mitigation.postprocessing.kamiranpost
   debiased.mitigation.postprocessing.pleiss
   debiased.mitigation.postprocessing.snel