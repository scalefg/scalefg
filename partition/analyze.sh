if [ -z "$3" ]
then
	echo "Usage: analyze.sh <method> <dataset> <num_parts>"
	exit
fi

echo "analyzing..."
./dgl_parse.py --method $1 --dataset $2 --num_parts $3
./analyze.py $2 $3 $1
rm -rf dataset/$2/dgl

echo "finished!"
