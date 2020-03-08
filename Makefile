.DEFAULT_GOAL : all
.PHONY : test testx profile%

IMAGEVIEW = gpicview

doc/index.html: epydoc.conf clmat.py
	epydoc --html --config=$< -v

test: unittests.py
	python -m pytest -r fEx $<

testx: unittests.py
	python -m pytest $< --assert='plain' -v -s -x -r fEx


PROFILE_ARGS = -f m+m -t 1000 -c -A 2048x2048

pycallgraph.png: testnumeric.py clmat.py
	pycallgraph graphviz -- ./$< ${PROFILE_ARGS}

profile: pycallgraph.png
	${IMAGEVIEW} $<

profile.dat: testnumeric.py clmat.py
	python -m cProfile -o $@ ./$< ${PROFILE_ARGS}

profile.png: profile.dat
	gprof2dot -f pstats $< | dot -Tpng -o $@

profile1: profile.png
	${IMAGEVIEW} $<

profile2: profile.dat
	pyprof2calltree -i $< -k

all: doc/index.html

clean:
	rm -f profile.* profile.png pycallgraph.png
