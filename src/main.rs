use rand::Rng;

fn main() {

    let sizes = vec![5,10,15,20];

    let mut arrays: Vec<Vec<usize>> = Vec::new();

    for &size in &sizes{
        arrays.push(generate_random_array(size, 1,100));
    }

    for (index, arr) in arrays.iter_mut().enumerate(){
        println!("Array {} (size: {}) before sorting: {:?}", index + 1, sizes[index], arr);
        selection_sort(arr);
        println!("Array after sorting: {:?}", arr);
        println!("________________________________________________");
    }

}

fn generate_random_array(size: usize, min: usize, max: usize) -> Vec<usize>{
    let mut rng = rand::thread_rng();
    let mut arr = Vec::with_capacity(size);

    for _ in 0..size {
        arr.push(rng.gen_range(min..=max));
    }

    arr
}

/*

Selection sort take an index
- look at what is after
- find the smallest 
- swap it with the i-th

It goes from 0 to n X i to n.

Cost of n(n+1)/2 ~ n^2
*/



fn selection_sort(arr: &mut [usize]) {
    let n = arr.len();

    for i in 0..n {
        let mut minimum: usize = usize::pow(2, 64 - 1); // set the max size for int as possible
        let mut min_pos: usize = 0;

        for j in i..n {
            if arr[j] < minimum {
                minimum = arr[j];
                min_pos = j;
            }
        }

        arr.swap(i, min_pos);
    }
}
