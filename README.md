# Численные методы линейной алгебры
*Программный проект студента ФКН НИУ ВШЭ (2020/2021). Библиотека с реализацией численных алгоритмов в линейной алгебре*

### Использованные инструменты разработки

* Язык `C++20`, компилятор `GCC 9.3.0`
* Библиотека линейной алгебры `Eigen 3.3.9`
* Система проверки формата `ClangFormat`. Код соответствует `Google C++ Style Guide`
* Система поддержки версий `Git`, инструмент `Git Submodules`

### Сборка проекта

**Для сборки необходимы:**

* Компилятор с поддержкой стандарта `C++20`
* Система сборки `CMake` версии 3.18.0 или выше

**Как собрать проект:**

*Шаг 1.* Добавление библиотеки [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) в рабочий каталог в качестве `Git Submodule`:

``
git clone --recurse-submodules
git submodule update --init --recursive
``

*Шаг 2.* Создание конфигурации с использованием CMake:

``
cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Release -H. -B./build -G "Unix Makefiles"
``

*Шаг 3.* Сборка с использованием CMake:

``
cmake --build ./build --config --target all -j 10 --
``
