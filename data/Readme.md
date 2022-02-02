# Original Data from November 2021
Original data for the experiments run over the Project X timeline can be found at the [here](https://drive.google.com/file/d/1XCg9wQ4xYP97TUzgiOKpriM7LEt4SpBy/view?usp=sharing).
Please email ciaran.bylesho@gmail.com for any questions

# Label Data Sources
The related code, visuals and datasets related to the Project X research project in 2022.

![image](https://user-images.githubusercontent.com/85202161/152131169-aa5a24c0-7856-4e2b-88dd-d66648b7e88f.png)
### The label data for LUAD was retrieved from [The Disease Gene Network Portal (DisGeNet)]() in November 2021.

The [Summary of GDAs](https://www.disgenet.org/browser/0/1/0/C0152013/) contains information about the Evidence Index and other types of GDA's.

The [Evidences of GDAs](https://www.disgenet.org/browser/0/1/1/C0152013/_a/_b./) contains information about the GDA score, the sentence supporting the association and other types of biomarkers that help structure the labels used to train the model.

<br /> 

# Node/Gene Data Sources
![image](https://user-images.githubusercontent.com/85202161/152130836-d449353b-b43f-423d-9508-050efa1baf2c.png)


### In November 2021, Node data and features were retrieved from [TCGA-LUAD project](https://portal.gdc.cancer.gov/exploration?filters=%7B%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-LUAD%22%5D%7D%2C%22op%22%3A%22in%22%7D%5D%2C%22op%22%3A%22and%22%7D&genesTable_offset=21000&genesTable_size=100&searchTableTab=genes).

### Despite being public data, NIH dataset cannot be downloaded in one click. **(it is technically possible but not with all the features such as # of cases in the cohort and number of mutations in the GDC portal).**

<br /> 


![image](https://user-images.githubusercontent.com/85202161/152131076-8a7daea8-a65b-491d-8b5c-d0ab55805bd8.png)

### In November 2021, Node data and features were also retrieved from the [Human Protein Atlas](https://www.proteinatlas.org/).


#### Conversions from gene ID to ensembl ID are available from [g:profiler](https://biit.cs.ut.ee/gprofiler/convert).


<br /> 


# Edges/Protein Data Sources

[![image](https://user-images.githubusercontent.com/85202161/152131328-7ccf9022-bf9d-4d5e-9393-9166782513e2.png)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fversion-11-0.string-db.org%2F&psig=AOvVaw1F6YMs8HO-yBu-Vne2Yp5X&ust=1643881960078000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCOjuz47h4PUCFQAAAAAdAAAAABAD)

### In November 2021, the protein data and edge features were retrieved from [STRING](https://string-db.org/cgi/download?sessionId=bjDATTcUSCjE&species_text=Homo+sapiens)
It is important to note we applied a homo sapien filter, and chose the following file: *9606.protein.links.detailed.v11.5.txt.gz (115.5 Mb)*
