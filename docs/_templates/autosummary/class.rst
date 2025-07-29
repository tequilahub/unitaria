{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% set public_members = (attributes + methods) | reject("in", ["__init__"]) | list %}

   {% block attributes %}
   {% if attributes %}
   {% for item in attributes %}
   {% if item not in inherited_members %}
   .. autoattribute:: {{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   {% for item in methods %}
   {%- if item != "__init__" %}
   {% if item not in inherited_members %}
   .. automethod:: {{ name }}.{{ item }}
   {% endif %}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block private %}
   {% if name == "Node" %}
   .. automethod:: Node._normalization
   .. automethod:: Node._subspace_in
   .. automethod:: Node._subspace_out
   .. automethod:: Node._circuit
   {% endif %}
   {% endblock %}

   {% block inherited %}
   {% set inherited = inherited_members | select("in", public_members) | list %}
   {% if inherited %}
   .. rubric:: Inherited members

   {% if attributes %}
   {% for item in attributes %}
   {% if item in inherited %}
   .. autoattribute:: {{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}

   {% if methods %}
   {% for item in methods %}
   {% if item in inherited %}
   .. automethod:: {{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}

   {% endif %}
   {% endblock %}
