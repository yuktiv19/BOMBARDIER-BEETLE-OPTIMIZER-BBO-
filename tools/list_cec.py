import opfunu

funcs = opfunu.get_cec_based_functions()
print('Total CEC-based functions found:', len(funcs))
count = 0

for it in funcs:
    try:
        if isinstance(it, tuple) and len(it) == 2:
            name, cls = it
        else:
            cls = it
            name = getattr(cls, '__name__', str(cls))

        module_name = getattr(cls, '__module__', '')

        if '2017' in name.lower() or '2017' in module_name.lower():
            print(f"{name} -> {module_name}")
            count += 1
    except Exception as e:
        print('Skipping item due to error:', e)

print('Filtered count:', count)
