if [ -z "$3" ]
then
	echo "Usage: run.sh <method> <dataset> <num_parts> [export]"
	exit
fi

case "$1" in
	metis)
		echo "using metis..."
		gpmetis -ncut=1 -niter=1 dataset/$2/$2.graph $3
		mv dataset/$2/$2.graph.part.$3 dataset/$2/$2.parts.$3.metis
		;;
	g3)
		echo "using g3 partitioner..."
		./gpmetis -ptype=rb -ncut=2 -niter=20 dataset/$2/$2.graph $3
		f="dataset/$2/$2.parts.$3"	
		mv dataset/$2/$2.graph.part.$3 $f
		./iterative_partitioning.py $2 $3
		rm $f
		;;
	*)
		./random_chunk.py --method $1 --dataset $2 --num_parts $3
		;;
esac

if [ ! -z "$4" ]
then
	echo "export to dgl partitions..."
	./dgl_parse.py --method $1 --dataset $2 --num_parts $3
	rm -rf /glusterfs/dataset/$2.$1.$3
	mv dataset/$2/dgl /glusterfs/dataset/$2.$1.$3
	#./analyze.py $2 $3 $1
	#rm -rf dataset/$2/dgl
fi

echo "finished!"
#gpmetis -ptype=rb -ncut=1 -niter=20 data/$1.graph $2
#gpmetis -ptype=kway -objtype=vol -ncut=1 -niter=20 data/$1.graph $2
#gpmetis -ncut=1 -niter=1 data/$1.graph $2
#./partition_graph.py --dataset $1 --num_parts $2
#./pd.py $1 $2 g3
