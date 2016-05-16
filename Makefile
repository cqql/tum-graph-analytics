include project.mk

.PHONY: all
all: $(BOSEN_LIBS)

.PHONY: clean
clean:
	rm -rf $(BOSEN_THIRD_PARTY_ROOT) $(BOSEN_ROOT) $(STRADS_ROOT)

### Libraries ###

$(BOSEN_THIRD_PARTY_ROOT):
	git clone $(BOSEN_THIRD_PARTY_REPO) $(BOSEN_THIRD_PARTY_ROOT)

$(BOSEN_THIRD_PARTY_LIBS): $(BOSEN_THIRD_PARTY_ROOT)
	make -j 5 -f $(BOSEN_THIRD_PARTY_ROOT)/Makefile

$(BOSEN_ROOT): $(BOSEN_THIRD_PARTY_LIBS)
	git clone $(BOSEN_REPO) $(BOSEN_ROOT)

$(BOSEN_LIBS): $(BOSEN_ROOT)
	cp --update $(BOSEN_ROOT)/defns.mk.template $(BOSEN_ROOT)/defns.mk
	sed --in-place --expression="s#PETUUM_THIRD_PARTY = \$$(PETUUM_ROOT)/third_party#PETUUM_THIRD_PARTY = $(BOSEN_THIRD_PARTY_ROOT)#g" $(BOSEN_ROOT)/defns.mk
	make -j 5 -f $(BOSEN_ROOT)/Makefile

$(STRADS_ROOT):
	git clone $(STRADS_REPO) $(STRADS_ROOT)
