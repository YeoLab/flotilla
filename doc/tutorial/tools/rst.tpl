{%- extends 'rst.tpl' -%}

{% block data_text scoped %}
{{ output.data['text/plain']}}
{% endblock data_text %}