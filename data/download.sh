for task_id in 1 2 3 4 5 6 7
do
	mkdir LaMP_${task_id}
	cd LaMP_${task_id}
	for _split in train dev
	do
		wget https://ciir.cs.umass.edu/downloads/LaMP/LaMP_${task_id}/${_split}/${_split}_questions.json
		wget https://ciir.cs.umass.edu/downloads/LaMP/LaMP_${task_id}/${_split}/${_split}_outputs.json
	done
	wget https://ciir.cs.umass.edu/downloads/LaMP/LaMP_${task_id}/test/test_questions.json
	cd ../
done