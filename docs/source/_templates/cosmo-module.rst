{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}
   :members:
   :exclude-members: Cosmology

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

.. currentmodule:: {{ fullname }}

.. autoclass:: Cosmology
   :members:
   :inherited-members:

   .. rubric:: Attributes

   .. autocosmosummary:: {{ fullname }}.Cosmology
       :attributes:

   .. rubric:: Methods

   .. autocosmosummary:: {{ fullname }}.Cosmology
       :methods:
