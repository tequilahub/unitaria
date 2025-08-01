{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if name == "Node" %}
.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :inherited-members:
   :exclude-members: _abc_impl
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
