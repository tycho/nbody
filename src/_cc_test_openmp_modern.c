extern int foo(int x);

int main() {
	unsigned int x, y = 0;
#pragma omp parallel for
	for (x = 0; x < 100; x++)
		    foo(x);
//#pragma omp atomic update
	y += x;
	return 0;
}
