import hypothesis

hypothesis.settings.register_profile("default", deadline=None)
hypothesis.settings.register_profile("ci", max_examples=100, deadline=None)
hypothesis.settings.register_profile("more", max_examples=1000, deadline=None)
hypothesis.settings.load_profile("default")
