.. _citation:

---------------
Citing conn2res
---------------

We're thrilled you've found ``conn2res`` useful in your work! Please cite the
following when referring to your use of the toolbox:

.. .. raw:: html

..    <script language="javascript">
..    var version = 'latest';
..    function fillCitation(){
..       $('#conn2res_version').text(version);

..       function cb(err, zenodoID) {
..          getCitation(zenodoID, 'vancouver-brackets-no-et-al', function(err, citation) {
..             $('#abagen_citation').text(citation);
..          });
..          getDOI(zenodoID, function(err, DOI) {
..             $('#abagen_doi_url').text('https://doi.org/' + DOI);
..             $('#abagen_doi_url').attr('href', 'https://doi.org/' + DOI);
..          });
..       }

..       if(version == 'latest') {
..          getLatestIDFromconceptID("3451463", cb);
..       } else {
..          getZenodoIDFromTag("3451463", version, cb);
..       }
..    }
..    </script>

..    <p style="margin-left: 30px">
..      <span id="abagen_citation">abagen</span> Available from: <a id="abagen_doi_url" href="https://doi.org/10.5281/zenodo.3451463">10.5281/zenodo.3451463</a>.
..      <img src onerror='fillCitation()' alt="" />
..    </p>
..    <p style="margin-left: 30px">
..      2. Arnatkevic̆iūtė, Aurina, Fulcher, Ben D., Fornito, Alex (2019). A practical guide to linking brain-wide gene expression and neuroimaging data. NeuroImage, 189, 353-367. doi:10.1016/j.neuroimage.2019.01.011
..    </p>
..    <p style="margin-left: 30px">
..      3. Hawrylycz, Michael J., Lein, Ed S., Guillozet-Bongaarts, Angela L., Shen, Elaine H., Ng, Lydia, Miller, Jeremy A., … Jones, Allan R. (2012). An anatomically comprehensive atlas of the adult human brain transcriptome. Nature, 489(7416), 391–399. doi:10.1038/nature11405
..    </p>

For more information about why citing software is important please refer to
`this article <https://www.software.ac.uk/how-cite-software>`_ from the
Software Sustainability Institute.

.. _DOI: https://en.wikipedia.org/wiki/Digital_object_identifier
.. _Zenodo: https://zenodo.org
