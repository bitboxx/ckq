// Test file for chunk depth visualization
// This file has chunks that end and start at the same line

pub mod first_module {
    pub fn function1() {
        println!("First module");
    }
}
pub mod second_module {
    pub fn function2() {
        println!("Second module");
    }
}

pub struct OuterStruct {
    field: i32,
}
impl OuterStruct {
    fn method1() {}
}
impl OuterStruct {
    fn method2() {}
}

pub mod nested {
    pub struct Inner {
        value: String,
    }

    impl Inner {
        pub fn new() -> Self {
            Self { value: String::new() }
        }

        pub fn process(&self) {
            if self.value.is_empty() {
                println!("Empty");
            }
        }
    }
}
