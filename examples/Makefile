test:

	python ipnbdoctest.py *.ipynb

pat = 'Figure at 0x[a-f0-9]*>'
rep = 'Figure at 0xFFFFFFFFF>'
sed = sed s/$(pat)/$(rep)/ | sed s/}\n\r/}/
dir = .

hexstrip:

	for ipynb in $(ls $(dir)/*.ipynb) ; do
		cat $ipynb | $(sed) > $ipynb.sed
		mv $ipynb.sed $ipynb
	done
