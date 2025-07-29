{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

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

   {% block inherited %}
   {% if inherited_members %}
   .. rubric:: Inherited members

   {% if attributes %}
   {% for item in attributes %}
   {% if item in inherited_members %}
   .. autoattribute:: {{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}

   {% if methods %}
   {% for item in methods %}
   {% if item in inherited_members %}
   .. automethod:: {{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}

   {% endif %}
   {% endblock %}
