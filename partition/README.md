# Graph partition process
1. Use export.py to download the original graph and export to pure text form.
	- available datasets: cora, reddit, obg-series
	- download amazon [here](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz)
2. Use run.sh to partition the graph
	- Usage `run.sh <method> <dataset> <num_parts>`
3. Refer to analyze.py on how to read the partitioning result into DGL format.
