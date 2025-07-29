{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if name == "Node" %}
.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:
   :private-members:
   :member-order: groupwise
{% else %}
.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:
   :member-order: groupwise
{% endif %}
