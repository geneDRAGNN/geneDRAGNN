##### projectx Queen's
# Label Data Sources
The related code, visuals and datasets related to the Queen's University Project X research project in 2022.

### The label data for LUAD was retrieved from [The Disease Gene Network Portal (DisGeNet)]() in November 2021.

The [Summary of GDAs](https://www.disgenet.org/browser/0/1/0/C0152013/) contains information about the Evidence Index and other types of GDA's.

The [Evidences of GDAs](https://www.disgenet.org/browser/0/1/1/C0152013/_a/_b./) contains information about the GDA score, the sentence supporting the association and other types of biomarkers that help structure the labels used to train the model.

## GDA Score and Evidence Index Calculations

![(0 6 if Nsources,  2](https://user-images.githubusercontent.com/85202161/152104621-bb9c6b54-1d1f-4cf3-9bf2-655fbd3092a8.png)
![supporting the genevariant-disease associations  This index is computed for the sources](https://user-images.githubusercontent.com/85202161/152104635-92c67ca8-e216-4e02-952f-fd54f27cceff.png)

## Node/Gene Data Sources
### In November 2021, Node data and features were retrieved from [TCGA-LUAD project](https://portal.gdc.cancer.gov/exploration?filters=%7B%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-LUAD%22%5D%7D%2C%22op%22%3A%22in%22%7D%5D%2C%22op%22%3A%22and%22%7D&genesTable_offset=21000&genesTable_size=100&searchTableTab=genes).

### The NIH does not permit one to download the entire dataset in one click  **(including the # of cases in the cohort and number of mutations in the GDC portal).**
We changed the settings in the NIH database to show 100 genes per page and manually downloaded each subset of 100 genes to compile the dataset into the form we used for aggregation.

#### Conversions from gene ID to ensembl ID are available from [g:profiler](https://biit.cs.ut.ee/gprofiler/convert).

### In November 2021, Node data and features were also retrieved from the [Human Protein Atlas](https://www.proteinatlas.org/).

## Edges/Protein Data Sources

### In November 2021, the protein data and edge features were retrieved from [STRING](https://string-db.org/cgi/download?sessionId=bjDATTcUSCjE&species_text=Homo+sapiens)
It is important to note we applied a homo sapien filter, and chose the following file: *9606.protein.links.detailed.v11.5.txt.gz (115.5 Mb)*


