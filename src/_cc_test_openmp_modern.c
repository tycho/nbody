extern int foo(int x);

int main() {
	unsigned int x;
#pragma omp parallel for
	for (x = 0; x < 100; x++)
		    foo(x);
	return 0;
}
